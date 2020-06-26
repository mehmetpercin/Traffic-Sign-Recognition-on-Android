package com.mp.camera2app;

import androidx.appcompat.app.ActionBar;
import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        ActionBar actionBar=getSupportActionBar();
        actionBar.hide();

        if(savedInstanceState==null){
            getSupportFragmentManager().beginTransaction().
                    replace(R.id.container,Camera2BasicFragment.newInstance()).commit();
        }
    }
}
